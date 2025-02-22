export type Probability = number[];

export interface TrainingHistory {
  epoch: number;
  loss: number;
  weights: number[];
}

export interface BackendResponse {
  probabilities: Probability[];
  trainingHistory: TrainingHistory[];
}