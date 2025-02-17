"use client";
import { useState, useEffect } from "react";

type Probability = number[];

interface TrainingHistory {
  epoch: number;
  loss: number;
  weights: number[];
}

interface BackendResponse {
  probabilities: Probability[];
  trainingHistory: TrainingHistory[];
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
      <div className="flex flex-col gap-5 justify-start">
        {backendResponse?.probabilities.map((probability, index) => (
          <div className="flex flex-col gap-1" key={index}>
            <div className="flex gap-2">
              <div className="font-bold">{`Probability ${index + 1}:`}</div>
              <div>{`${probability.map(
                (probability) => `${probability} `
              )}`}</div>
            </div>
            <div className="flex gap-2">
              <div className="font-bold">Loss: </div>
              <div>{backendResponse?.trainingHistory[index].loss}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
