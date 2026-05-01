import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { apiFetch, readApiError } from "../api/http";

export type User = {
  id: string;
  email: string;
  full_name: string | null;
  onboarding_completed: boolean;
};

type AuthState = {
  accessToken: string | null;
  refreshToken: string | null;
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
  refreshAccess: () => Promise<boolean>;
  setUser: (u: User | null) => void;
};

const AuthContext = createContext<AuthState | null>(null);

const ACCESS = "stp_access";
const REFRESH = "stp_refresh";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [accessToken, setAccessToken] = useState<string | null>(() => localStorage.getItem(ACCESS));
  const [refreshToken, setRefreshToken] = useState<string | null>(() => localStorage.getItem(REFRESH));
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const persistTokens = useCallback((a: string, r: string) => {
    localStorage.setItem(ACCESS, a);
    localStorage.setItem(REFRESH, r);
    setAccessToken(a);
    setRefreshToken(r);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(ACCESS);
    localStorage.removeItem(REFRESH);
    setAccessToken(null);
    setRefreshToken(null);
    setUser(null);
  }, []);

  const refreshAccess = useCallback(async (): Promise<boolean> => {
    const rt = refreshToken || localStorage.getItem(REFRESH);
    if (!rt) return false;
    try {
      const res = await apiFetch("/api/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: rt }),
        auth: false,
      });
      if (!res.ok) return false;
      const data = (await res.json()) as { access_token: string; refresh_token: string };
      persistTokens(data.access_token, data.refresh_token);
      return true;
    } catch {
      return false;
    }
  }, [persistTokens, refreshToken]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const at = accessToken || localStorage.getItem(ACCESS);
      if (!at) {
        setLoading(false);
        return;
      }
      const res = await apiFetch("/api/auth/me", { auth: true, token: at });
      if (cancelled) return;
      if (res.status === 401) {
        const ok = await refreshAccess();
        if (cancelled) return;
        if (ok) {
          const at2 = localStorage.getItem(ACCESS);
          if (at2) {
            const r2 = await apiFetch("/api/auth/me", { auth: true, token: at2 });
            if (r2.ok) setUser((await r2.json()) as User);
            else logout();
          }
        } else logout();
      } else if (res.ok) {
        setUser((await res.json()) as User);
      } else {
        logout();
      }
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await apiFetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        auth: false,
      });
      if (!res.ok) {
        throw new Error(await readApiError(res));
      }
      const data = (await res.json()) as {
        tokens: { access_token: string; refresh_token: string };
        user: User;
      };
      persistTokens(data.tokens.access_token, data.tokens.refresh_token);
      setUser(data.user);
    },
    [persistTokens]
  );

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      const res = await apiFetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, full_name: fullName || null }),
        auth: false,
      });
      if (!res.ok) {
        throw new Error(await readApiError(res));
      }
      const data = (await res.json()) as {
        tokens: { access_token: string; refresh_token: string };
        user: User;
      };
      persistTokens(data.tokens.access_token, data.tokens.refresh_token);
      setUser(data.user);
    },
    [persistTokens]
  );

  const value = useMemo(
    () => ({
      accessToken,
      refreshToken,
      user,
      loading,
      login,
      register,
      logout,
      refreshAccess,
      setUser,
    }),
    [accessToken, refreshToken, user, loading, login, register, logout, refreshAccess]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
