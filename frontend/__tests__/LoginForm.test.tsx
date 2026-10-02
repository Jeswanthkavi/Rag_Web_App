import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LoginForm } from "../src/components/auth/LoginForm";
import { AuthProvider } from "../src/context/AuthContext";
import { api } from "../src/lib/api";

// Mock next/router
jest.mock("next/router", () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
  }),
}));

describe("LoginForm Component", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders email and password inputs and submit button", () => {
    render(
      <AuthProvider>
        <LoginForm />
      </AuthProvider>
    );

    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("submits valid credentials and calls login API", async () => {
    const mockLogin = jest.spyOn(api, "login").mockResolvedValueOnce({
      access_token: "fake_token_123",
      token_type: "bearer",
      expires_in: 3600,
    });
    jest.spyOn(api, "getMe").mockResolvedValueOnce({
      id: "u1",
      email: "test@example.com",
      is_active: true,
      is_superuser: false,
      created_at: "2026-01-01",
      updated_at: "2026-01-01",
    });

    render(
      <AuthProvider>
        <LoginForm />
      </AuthProvider>
    );

    fireEvent.change(screen.getByLabelText(/email address/i), {
      target: { value: "test@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: "Password123!" },
    });

    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("test@example.com", "Password123!");
    });
  });
});
