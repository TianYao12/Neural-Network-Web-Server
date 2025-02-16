"use client";
import { useState, useEffect } from "react";

interface BackendResponse {
  message: string;
}

export default function Home() {
  const [backendResponse, setBackendResponse] =
    useState<BackendResponse | null>(null);
  useEffect(() => {
    const testRequestToBackend = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}`, {
          method: "GET",
        });
        const result = await response.json();
        setBackendResponse(result);
        console.log(result);
      } catch (error) {
        console.error(error);
      }
    };
    testRequestToBackend();
  }, []);
  return (
    <div className="mt-32 flex flex-col gap-10 w-full justify-center items-center font-[family-name:var(--font-geist-sans)]">
      <h1 className="text-3xl">Neural Network Visualization</h1>
      <p>{"Backend Response: " + backendResponse?.message}</p>
    </div>
  );
}
