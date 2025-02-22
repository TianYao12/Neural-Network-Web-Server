"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TrainingHistory } from "@/lib/types";

const LossChart = ({
  trainingHistory,
}: {
  trainingHistory: TrainingHistory[];
}) => {
  return (
    <div className="w-full max-w-2xl pb-16">
      <h2 className="text-xl font-bold mb-4">Loss Over Epochs</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={trainingHistory}>
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
            label={{ value: "Loss", angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#8884d8"
            strokeWidth={3}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LossChart;
