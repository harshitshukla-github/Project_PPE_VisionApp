// In-memory stand-in for expo-secure-store (jest has no Keychain/Keystore).
// Node-module mocks next to node_modules apply automatically in every test.
const store = new Map();

module.exports = {
  getItemAsync: jest.fn(async (key) => (store.has(key) ? store.get(key) : null)),
  setItemAsync: jest.fn(async (key, value) => {
    store.set(key, value);
  }),
  deleteItemAsync: jest.fn(async (key) => {
    store.delete(key);
  }),
  isAvailableAsync: jest.fn(async () => true),
  __store: store,
};
