/**
 * Authentication Context Provider.
 * Handles token initialization, user session lifecycle, login, registration, and logout.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { api } from "../lib/api";
import { APIError, User } from "../types";

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadUser = useCallback(async () => {
    try {
      const storedToken =
        typeof window !== "undefined"
          ? localStorage.getItem("rag_auth_token")
          : null;

      if (!storedToken) {
        setUser(null);
        setToken(null);
        setIsLoading(false);
        return;
      }

      setToken(storedToken);
      const currentUser = await api.getMe();
      setUser(currentUser);
    } catch {
      // Stored token is invalid or expired
      api.logout();
      setUser(null);
      setToken(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  const login = async (email: string, password: string): Promise<void> => {
    setIsLoading(true);
    setError(null);
    try {
      const tokenResp = await api.login(email, password);
      setToken(tokenResp.access_token);
      const currentUser = await api.getMe();
      setUser(currentUser);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      const msg = apiErr.detail || "Authentication failed. Please verify your credentials.";
      setError(msg);
      throw apiErr;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, password: string): Promise<void> => {
    setIsLoading(true);
    setError(null);
    try {
      await api.register(email, password);
      // Auto-login after successful registration
      await login(email, password);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      const msg = apiErr.detail || "Registration failed. Please try again.";
      setError(msg);
      throw apiErr;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
    setToken(null);
    setError(null);
  }, []);

  const clearError = () => setError(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user && !!token,
        isLoading,
        error,
        login,
        register,
        logout,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
