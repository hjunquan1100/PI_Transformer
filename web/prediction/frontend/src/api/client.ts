import axios, { AxiosError } from "axios";
import { ElMessage } from "element-plus";

export const client = axios.create({
  baseURL: "",
  timeout: 120000,
});

export function showApiError(err: unknown, fallback = "Request failed") {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError<{ detail?: string }>;
    const detail = ax.response?.data?.detail;
    const msg = typeof detail === "string" ? detail : fallback;
    ElMessage.error(msg);
    return;
  }
  ElMessage.error(fallback);
}

export function structureSvgUrl(smiles: string): string {
  return `/api/structure/svg?smiles=${encodeURIComponent(smiles)}`;
}

/** Structure PNG with embedded SMILES metadata. */
export function structurePngUrl(smiles: string): string {
  return `/api/structure/png?smiles=${encodeURIComponent(smiles)}`;
}
