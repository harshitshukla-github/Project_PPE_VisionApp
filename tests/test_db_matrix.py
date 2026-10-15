"""The job matrix in a real PostgreSQL database.

Runs against TEST_DATABASE_URL, which `cli.py db setup` writes to .env and
CI provides. Skipped when it is not set. Every test starts from an empty
schema, so the database named there is wiped: it must be named *_test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import Engine, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.engine import get_engine
from src.db.matrix import ITEMS, read_matrix, seed_from_yaml
from src.db.migrate import alembic_config, current_revision, head_revision, upgrade_to_head
from src.db.models import Job, JobRequirement, PpeItem
from src.inference.compliance import (
    check_compliance,
    invalidate_job_matrix_cache,
    load_job_matrix,
)
from src.settings import settings
from tests.conftest import make_detection

YAML_PATH = Path("configs/job_matrix.yaml")
URL = settings.test_database_url

pytestmark = pytest.mark.skipif(
    not URL, reason="TEST_DATABASE_URL not set — run `cli.py db setup`"
)


def _yaml_matrix() -> dict[str, dict[str, list[str]]]:
    return {
        job: {"required": req.required, "pending": req.pending}
        for job, req in load_job_matrix(path=YAML_PATH).items()
    }


@pytest.fixture
def empty_db() -> Engine:
    """An engine on the test database, with every table dropped."""
    dbname = URL.rsplit("/", 1)[-1].split("?", 1)[0]
    if not dbname.endswith("_test"):
        pytest.fail(f"refusing to wipe '{dbname}': TEST_DATABASE_URL must name a *_test database")
    engine = get_engine(URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    return engine


@pytest.fixture
def migrated_db(empty_db: Engine) -> Engine:
    upgrade_to_head(URL)
    return empty_db


@pytest.fixture
def seeded_db(migrated_db: Engine) -> Engine:
    seed_from_yaml(migrated_db, YAML_PATH)
    return migrated_db


class TestMigrations:
    def test_upgrade_reaches_head(self, migrated_db: Engine) -> None:
        assert current_revision(URL) == head_revision()

    def test_models_and_migrations_agree(self, migrated_db: Engine) -> None:
        """Fails when a model changes without a migration (or the reverse)."""
        command.check(alembic_config(URL))

    def test_downgrade_and_upgrade_again(self, migrated_db: Engine) -> None:
        command.downgrade(alembic_config(URL), "base")
        assert current_revision(URL) is None
        upgrade_to_head(URL)
        assert current_revision(URL) == head_revision()


class TestSeed:
    def test_database_matrix_matches_the_yaml_exactly(self, seeded_db: Engine) -> None:
        """Same jobs, same lists, same order: the app must see no difference."""
        assert list(read_matrix(seeded_db).items()) == list(_yaml_matrix().items())

    def test_every_item_is_catalogued(self, seeded_db: Engine) -> None:
        with Session(seeded_db) as session:
            items = {i.key: i.detectable for i in session.scalars(select(PpeItem))}
        assert items == {key: detectable for key, _, detectable in ITEMS}

    def test_second_seed_changes_nothing(self, seeded_db: Engine) -> None:
        before = read_matrix(seeded_db)
        report = seed_from_yaml(seeded_db, YAML_PATH)
        assert (report.items_added, report.jobs_added) == (0, 0)
        assert "source of truth" in report.skipped_reason
        assert read_matrix(seeded_db) == before

    def test_seed_never_overwrites_edits(self, seeded_db: Engine) -> None:
        with Session(seeded_db) as session, session.begin():
            painter = session.scalar(select(Job).where(Job.key == "painter"))
            session.add(JobRequirement(job_id=painter.id, item_key="boots", mode="required",
                                       position=99))
        seed_from_yaml(seeded_db, YAML_PATH)
        assert "boots" in read_matrix(seeded_db)["painter"]["required"]

    def test_unknown_item_in_yaml_is_rejected(self, migrated_db: Engine, tmp_path: Path) -> None:
        bad = tmp_path / "matrix.yaml"
        bad.write_text("jobs:\n  diver:\n    required: [flippers]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="flippers"):
            seed_from_yaml(migrated_db, bad)


class TestReading:
    def test_inactive_jobs_are_left_out(self, seeded_db: Engine) -> None:
        with Session(seeded_db) as session, session.begin():
            session.execute(update(Job).where(Job.key == "welder").values(is_active=False))
        assert "welder" not in read_matrix(seeded_db)

    def test_position_sets_the_order(self, seeded_db: Engine) -> None:
        with Session(seeded_db) as session, session.begin():
            session.execute(update(Job).where(Job.key == "painter").values(position=-1))
        assert next(iter(read_matrix(seeded_db))) == "painter"


class TestConstraints:
    """The database itself refuses data the API would choke on."""

    @pytest.mark.parametrize("key", ["Welder", "1st_job", "weld-er", "weld er", ""])
    def test_job_keys_must_be_lowercase_identifiers(
        self, migrated_db: Engine, key: str
    ) -> None:
        with pytest.raises(IntegrityError), Session(migrated_db) as session, session.begin():
            session.add(Job(key=key, display_name="x"))

    def test_duplicate_job_key_is_rejected(self, seeded_db: Engine) -> None:
        with pytest.raises(IntegrityError), Session(seeded_db) as session, session.begin():
            session.add(Job(key="welder", display_name="Welder again"))

    def test_mode_must_be_required_or_pending(self, seeded_db: Engine) -> None:
        with pytest.raises(IntegrityError), Session(seeded_db) as session, session.begin():
            job = session.scalar(select(Job).where(Job.key == "painter"))
            session.add(JobRequirement(job_id=job.id, item_key="boots", mode="optional"))

    def test_requirement_must_name_a_known_item(self, seeded_db: Engine) -> None:
        with pytest.raises(IntegrityError), Session(seeded_db) as session, session.begin():
            job = session.scalar(select(Job).where(Job.key == "painter"))
            session.add(JobRequirement(job_id=job.id, item_key="flippers", mode="required"))

    def test_deleting_a_job_deletes_its_requirements(self, seeded_db: Engine) -> None:
        with Session(seeded_db) as session, session.begin():
            session.delete(session.scalar(select(Job).where(Job.key == "welder")))
        with Session(seeded_db) as session:
            left = session.scalar(text(
                "SELECT count(*) FROM job_requirements r LEFT JOIN jobs j ON j.id = r.job_id "
                "WHERE j.id IS NULL"
            ))
        assert left == 0


class TestEndToEnd:
    @pytest.fixture
    def use_test_db(self, seeded_db: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
        monkeypatch.setattr(settings, "database_url", URL)
        return seeded_db

    def test_jobs_endpoint_is_unchanged_for_the_app(self, api_client, use_test_db) -> None:
        """GET /jobs from the database returns byte-for-byte what YAML did."""
        assert api_client.get("/api/v1/jobs").json() == _yaml_matrix()

    def test_an_edit_in_the_database_changes_verdicts(self, use_test_db: Engine) -> None:
        dets = [make_detection("hardhat"), make_detection("safety_vest")]
        assert check_compliance(dets, "general_labor").verdict.value == "NEEDS_REVIEW"

        with Session(use_test_db) as session, session.begin():
            job = session.scalar(select(Job).where(Job.key == "general_labor"))
            session.execute(
                JobRequirement.__table__.delete().where(
                    (JobRequirement.job_id == job.id) & (JobRequirement.item_key == "boots")
                )
            )
        invalidate_job_matrix_cache()
        assert check_compliance(dets, "general_labor").verdict.value == "PASS"
