import { client, showApiError } from "./client";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  decimer_available: boolean;
  device: string;
  inverse_model_loaded?: boolean;
  inverse_ckpt?: string | null;
}

export interface PredictResponse {
  smiles: string;
  tg_celsius: number;
  unit: string;
  parsed_from?: string;
  recognized_smiles?: string | null;
  matched_reference?: boolean;
}

export { structureSvgUrl, structurePngUrl } from "./client";

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/api/health");
  return data;
}

export async function predictFromSmiles(smiles: string): Promise<PredictResponse> {
  try {
    const { data } = await client.post<PredictResponse>("/api/predict/smiles", {
      smiles,
    });
    return data;
  } catch (err) {
    showApiError(err, "SMILES predictionfailed");
    throw err;
  }
}

export async function predictFromImage(
  file: File,
  referenceSmiles?: string
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  if (referenceSmiles && referenceSmiles.trim()) {
    form.append("reference_smiles", referenceSmiles.trim());
  }
  try {
    const { data } = await client.post<PredictResponse>(
      "/api/predict/image",
      form,
      { headers: { "Content-Type": "multipart/form-data" }, timeout: 180000 }
    );
    return data;
  } catch (err) {
    showApiError(err, "structure imagepredictionfailed");
    throw err;
  }
}
