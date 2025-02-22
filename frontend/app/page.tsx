"use client"

import { useEffect, useState } from "react";
import { BackendResponse } from "@/lib/types";
import LossChart from "@/components/LossChart";

export default function Home() {
  const [backendResponse, setBackendResponse] = useState<BackendResponse | null>(null);

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
    <div className="mt-32 flex flex-col gap-10 w-full justify-center items-center font-[family-name:var(--font-geist-sans)]">
      <h1 className="text-3xl">Neural Network Visualization</h1>
      <div className="flex flex-col gap-5 justify-start">
        {backendResponse?.probabilities.map((probability, index) => (
          <div className="flex flex-col gap-1" key={index}>
            <div className="flex gap-2">
              <div className="font-bold">{`Probability ${index + 1}:`}</div>
              <div>{`${probability.join(" ")}`}</div>
            </div>
            <div className="flex gap-2">
              <div className="font-bold">Loss: </div>
              <div>{backendResponse?.trainingHistory[index].loss}</div>
            </div>
          </div>
        ))}
      </div>
      {backendResponse?.trainingHistory && <LossChart trainingHistory={backendResponse.trainingHistory} />}
    </div>
  );
}
