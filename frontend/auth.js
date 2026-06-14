/* ─────────────────────────────────────────────────────────────────
   LexiScan Auto — Auth Helper (shared between login.html & index.html)
   ───────────────────────────────────────────────────────────────── */

const TOKEN_KEY = "lexiscan_token";
const USER_KEY  = "lexiscan_user";

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function getStoredUser() {
  const raw = localStorage.getItem(USER_KEY);
  return raw ? JSON.parse(raw) : null;
}

function setStoredUser(user) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function isLoggedIn() {
  return !!getToken();
}

function requireAuth() {
  if (!isLoggedIn()) {
    window.location.replace("/login");
  }
}

function logout() {
  clearToken();
  window.location.replace("/login");
}

/**
 * Authenticated fetch — automatically attaches Bearer token.
 * On 401, clears token and redirects to login.
 */
async function authFetch(url, options = {}) {
  const token = getToken();
  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    window.location.replace("/login");
    return res;
  }
  return res;
}
