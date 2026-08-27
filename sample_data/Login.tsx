import React, { useState } from "react";
import { api } from "../lib/api";

export interface LoginProps {
  onSuccess: (token: string) => void;
}

// 로그인 폼 컴포넌트
export function LoginForm({ onSuccess }: LoginProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async () => {
    const res = await api.post("/login", { email, password });
    if (res.ok) {
      onSuccess(res.token); // 부모에게 토큰 전달
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input value={email} onChange={(e) => setEmail(e.target.value)} />
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <button type="submit">로그인</button>
    </form>
  );
}

export const API_TIMEOUT = 5000;

export type AuthState = "idle" | "loading" | "done";

export default LoginForm;
