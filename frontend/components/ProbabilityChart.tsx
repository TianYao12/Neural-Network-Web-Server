"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { BackendResponse } from "@/lib/types";

const getClassColor = (index: number) => {
  const baseColors = [
    "#FF0000",
    "#FF7F00",
    "#E8E857",
    "#7FFF00",
    "#00FF00",
    "#00FF7F",
    "#00FFFF",
    "#007FFF",
    "#0000FF",
    "#990000"
  ];

  return baseColors[index] || `hsl(${(index * 40) % 360}, 70%, 50%)`;
};

const ProbabilityChart = ({
  backendResponse,
}: {
  backendResponse: BackendResponse;
}) => {
  const formattedData = backendResponse.probabilities.map(
    (probabilities, epochIndex) => {
      const dataPoint: { epoch: number } & Record<string, number> = {
        epoch: epochIndex + 1,
      };

      probabilities.forEach((prob, index) => {
        dataPoint[`Class ${index}`] = prob; 
      });

      return dataPoint;
    }
  );

  return (
    <div className="w-full max-w-2xl">
      <h2 className="text-xl font-bold mb-4">Probabilities Per Epoch</h2>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="epoch"
            label={{
              value: "Epoch",
              position: "insideBottomRight",
              offset: -5,
            }}
          />
          <YAxis
            label={{ value: "Probability", angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Legend />
          {backendResponse.probabilities[0].map((_, classIndex) => (
            <Bar
              key={classIndex}
              dataKey={`Class ${classIndex}`}
              fill={getClassColor(classIndex)}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ProbabilityChart;
