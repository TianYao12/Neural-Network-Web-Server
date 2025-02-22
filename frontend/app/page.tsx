"use client";

import { useEffect, useState } from "react";
import { BackendResponse } from "@/lib/types";
import LossChart from "@/components/LossChart";
import ProbabilityChart from "@/components/ProbabilityChart";

export default function Home() {
  const [backendResponse, setBackendResponse] =
    useState<BackendResponse | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}`, {
          method: "GET",
        });
        const result = await response.json();
        setBackendResponse(result);
      } catch (error) {
        console.error(error);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="pt-32 bg-gray-100 flex flex-col gap-10 w-full justify-center items-center font-[family-name:var(--font-geist-sans)]">
      <h1 className="text-3xl ">Neural Network Visualization</h1>
      {backendResponse?.probabilities && (
        <ProbabilityChart backendResponse={backendResponse} />
      )}
      {backendResponse?.trainingHistory && (
        <LossChart trainingHistory={backendResponse.trainingHistory} />
      )}
    </div>
  );
}
