import React, { useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { LoginForm } from "../components/auth/LoginForm";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.replace("/");
    }
  }, [isLoading, isAuthenticated, router]);

  if (isLoading) return null;

  return (
    <>
      <Head>
        <title>Login — Advanced RAG</title>
      </Head>
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem 1rem",
          background:
            "radial-gradient(ellipse at 50% 30%, rgba(99, 102, 241, 0.15), transparent 70%)",
        }}
      >
        <LoginForm />
      </div>
    </>
  );
}
