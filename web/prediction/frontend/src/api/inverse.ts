import { client, showApiError } from "./client";

export interface GeneratedMolecule {
  id: string;
  smiles: string;
  tg_target_c: number;
  pred_tg_c: number;
  tg_error_c: number;
  rank: number;
  valid: boolean;
}

export interface InverseGenerateResponse {
  tg_target_c: number;
  n_generated: number;
  valid_count: number;
  passed_count: number;
  recommended: GeneratedMolecule[];
  others: GeneratedMolecule[];
}

export async function generateInverse(
  tg_target_c: number
): Promise<InverseGenerateResponse> {
  try {
    const { data } = await client.post<InverseGenerateResponse>(
      "/api/inverse/generate",
      { tg_target_c },
      { timeout: 600000 }
    );
    return data;
  } catch (err) {
    showApiError(err, "inversegeneratefailed");
    throw err;
  }
}
