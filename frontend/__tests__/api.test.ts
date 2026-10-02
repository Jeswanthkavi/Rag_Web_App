import { api } from "../src/lib/api";

describe("API Client Abstraction", () => {
  beforeEach(() => {
    localStorage.clear();
    (global as any).fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("attaches Authorization header when JWT token is present", async () => {
    api.setAuthToken("test_jwt_token_xyz");

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        id: "u1",
        email: "test@example.com",
        is_active: true,
        is_superuser: false,
        created_at: "2026-01-01",
        updated_at: "2026-01-01",
      }),
    });

    const user = await api.getMe();
    expect(user.email).toBe("test@example.com");

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/users/me"),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer test_jwt_token_xyz",
        }),
      })
    );
  });

  it("handles 401 unauthenticated errors and clears token", async () => {
    api.setAuthToken("expired_token");

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: "Unauthorized",
      json: async () => ({ detail: "Invalid access token" }),
    });

    await expect(api.getMe()).rejects.toEqual({
      status: 401,
      detail: "Invalid access token",
    });

    expect(localStorage.getItem("rag_auth_token")).toBeNull();
  });
});
