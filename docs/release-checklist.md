# PPB Check — release checklist

The three open gates from the [roadmap](../agents.md#14-phased-roadmap).
Record the decision or verification evidence before checking an item off.

| Gate | Current status | Completion condition |
|---|---|---|
| Privacy review and app release (P3/P6) | Blocked on notice wording approval | The company privacy reviewer approves the wording, then Priyanka prepares the release build |
| Dashboard completion (P5) | Open: three decisions | All three decisions below are recorded and any resulting changes are implemented and verified |
| Required worker login (P7) | Built; cutover pending | The release build is on every phone in use, each site confirms it, and there are no unsigned checks for three working days in a row |

The release sequence is: privacy approval, Priyanka's release build, rollout
to every phone, then the P7 cutover once its observation condition is met.
The P5 decisions can be resolved while privacy review is pending; they are
the remaining gate for marking the dashboard phase finished.

## 1. Privacy review and app release

**No app release until the notice wording passes the company's privacy
check.** The notice and its review requirement are in
[checks-api.md §5](checks-api.md#5-consent-mobile-app) and
[§10, point 1](checks-api.md#10-decisions-on-the-review-questions).

- [ ] Any requested wording changes are applied to the contract and the
  server notice, with a new notice version as required by §5; record the
  approved version. Wording-only changes are server changes and need no
  app update: the app displays the server's `text` and `training_question`.
  If no wording change is needed, record that.
- [ ] The company privacy reviewer approves the exact notice wording.
  Record the reviewer, date and approval reference below.
- [ ] After approval, **Priyanka prepares the release build** with login,
  the photo notice, the name **PPB Check** and the new icon.
- [ ] Verify the installed release build: the name and icon, sign-in,
  first-login PIN change, photo notice, a check, and the training choice in
  Settings. Record the build identifier and verification result.

The release build is the first place the new name and icon appear on the
phone. Expo Go shows its own branding; the existing Expo Go phone tests do
not complete this release-build verification.

| Evidence | Record |
|---|---|
| Privacy reviewer, approval date and reference | Pending |
| Approved notice version / wording change reference | Pending |
| Release build identifier and verification result | Pending |

## 2. Dashboard decisions — P5

These are the **last three open P5 decisions**, for Harshit to decide.
Record each answer and its decision reference; none is decided here.

| Decision | Answer / reference |
|---|---|
| Should Reset password, Reset PIN and Deactivate ask for confirmation first? Record the answer for each action. | Pending |
| Can a dashboard user edit their own name? | Pending |
| What order should the job matrix save and display the requirements in? | Pending |

- [ ] All three answers are recorded.
- [ ] Any resulting contract changes go through a PR updating
  [admin-api.md](admin-api.md); implement and verify the agreed behavior.
- [ ] Mark P5 finished in the roadmap only after the decisions and any
  resulting work are complete.

## 3. Required worker login — P7

Keep `PPB_WORKER_AUTH=optional` until **all** readiness checks below pass.
This gate depends on the privacy approval and release build in item 1.
The cutover procedure remains [auth-api.md §5](auth-api.md#5-changes-to-existing-endpoints).

- [ ] Every phone in use has the release build from item 1. Each site's
  supervisor confirms this for their site; record the site confirmations
  and build identifier.
- [ ] No check without a login has reached the server for **three working
  days in a row**. Record the observation dates and evidence from the
  dashboard's unsigned-check count or server logs. A zero count alone does
  not replace the site confirmations.
- [ ] **Harshit (admin)** switches worker login to `required`, either in
  the dashboard's `worker_auth` setting or with
  `PPB_WORKER_AUTH=required` in `.env` and a server restart.
  Check the effective setting: a saved dashboard override wins over `.env`
  until cleared. The dashboard change is recorded in the audit log.
- [ ] Verify a signed-in check still works and a request without a token
  gets `401 not_authenticated`. Record the cutover date and evidence here
  and in auth-api.md §5.

**Rollback:** set the effective worker login setting back to `optional`,
using the dashboard if it holds an override. Login-capable phones continue
to work. Record any rollback and its reason before retrying the cutover.

| Evidence | Record |
|---|---|
| Release build and confirmation from each site | Pending |
| Three working days observed and unsigned-check evidence | Pending |
| Cutover date, effective setting and verification result | Pending |
| Rollback, if needed | None recorded |
