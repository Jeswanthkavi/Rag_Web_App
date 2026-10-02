import React, { useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { RegisterForm } from "../components/auth/RegisterForm";
import { useAuth } from "../context/AuthContext";

export default function RegisterPage() {
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
        <title>Register — Advanced RAG</title>
      </Head>
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem 1rem",
          background:
            "radial-gradient(ellipse at 50% 30%, rgba(6, 182, 212, 0.15), transparent 70%)",
        }}
      >
        <RegisterForm />
      </div>
    </>
  );
}
