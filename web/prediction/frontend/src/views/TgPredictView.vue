<template>
  <el-card class="predict-card" shadow="hover">
    <el-tabs v-model="activeTab" class="predict-tabs">
      <el-tab-pane label="SMILES prediction" name="smiles">
        <div class="panel" v-loading="smilesLoading">
          <p class="hint">
            Enter a polyimide repeat-unit SMILES. Connection points can be written as <code>*</code>.
          </p>
          <el-input
            v-model="smilesInput"
            type="textarea"
            :rows="5"
            placeholder="Example: *CC(=O)c1ccc(C(=O)C*)cc1*"
            spellcheck="false"
          />
          <div class="actions">
            <el-button @click="previewStructure" :disabled="!smilesInput.trim()">
              Show Structure
            </el-button>
            <el-button
              type="primary"
              @click="runSmilesPredict"
              :disabled="!smilesInput.trim()"
              :loading="smilesLoading"
            >
              Predict Tg
            </el-button>
          </div>
          <div v-if="structurePreviewUrl" class="structure-box">
            <img :src="structurePreviewUrl" alt="2D structure" />
          </div>
          <ResultPanel v-if="smilesResult" :result="smilesResult" />
        </div>
      </el-tab-pane>

      <el-tab-pane label="Structure Image Prediction" name="image">
        <div class="panel" v-loading="imageLoading">
          <p class="hint">
            Upload a structure image (PNG/JPEG). If it matches the SMILES input,
            the backend uses the reference SMILES so both prediction paths stay aligned.
          </p>
          <el-upload
            drag
            :auto-upload="false"
            :limit="1"
            accept="image/png,image/jpeg,image/jpg"
            :on-change="onFileChange"
            :on-remove="onFileRemove"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              Drop an image here or <em>click to upload</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">PNG/JPEG only, maximum 5 MB</div>
            </template>
          </el-upload>

          <div v-if="imagePreview" class="image-preview">
            <img :src="imagePreview" alt="uploaded structure" />
          </div>

          <el-input
            v-if="parsedSmiles"
            v-model="parsedSmiles"
            type="textarea"
            :rows="3"
            label="Recognized SMILES"
            class="parsed-smiles"
          />

          <div class="actions">
            <el-button
              type="primary"
              @click="runImagePredict"
              :disabled="!selectedFile"
              :loading="imageLoading"
            >
              Recognize and Predict
            </el-button>
            <el-button
              @click="repredictFromParsed"
              :disabled="!parsedSmiles.trim()"
              :loading="smilesLoading"
            >
              Predict with Current SMILES
            </el-button>
          </div>

          <div v-if="parsedSmiles && structureFromImageUrl" class="structure-box">
            <img :src="structureFromImageUrl" alt="2D structure from SMILES" />
          </div>

          <ResultPanel v-if="imageResult" :result="imageResult" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </el-card>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { UploadFilled } from "@element-plus/icons-vue";
import type { UploadFile, UploadFiles } from "element-plus";
import { ElMessage } from "element-plus";

import {
  predictFromImage,
  predictFromSmiles,
  structureSvgUrl,
  type PredictResponse,
} from "@/api/predict";
import ResultPanel from "@/components/ResultPanel.vue";

const activeTab = ref("smiles");
const smilesInput = ref("");
const smilesLoading = ref(false);
const smilesResult = ref<PredictResponse | null>(null);
const structurePreviewUrl = ref("");

const selectedFile = ref<File | null>(null);
const imagePreview = ref("");
const imageLoading = ref(false);
const imageResult = ref<PredictResponse | null>(null);
const parsedSmiles = ref("");

const structureFromImageUrl = computed(() =>
  parsedSmiles.value.trim() ? structureSvgUrl(parsedSmiles.value.trim()) : ""
);

function onFileChange(uploadFile: UploadFile, _files: UploadFiles) {
  const raw = uploadFile.raw;
  if (!raw) return;
  if (raw.size > 5 * 1024 * 1024) {
    ElMessage.warning("Image must not exceed 5 MB");
    return;
  }
  selectedFile.value = raw;
  imagePreview.value = URL.createObjectURL(raw);
  imageResult.value = null;
  parsedSmiles.value = "";
}

function onFileRemove() {
  selectedFile.value = null;
  imagePreview.value = "";
  parsedSmiles.value = "";
  imageResult.value = null;
}

function previewStructure() {
  const s = smilesInput.value.trim();
  if (!s) return;
  // Use SVG for preview and refresh the URL to avoid stale browser cache.
  structurePreviewUrl.value = structureSvgUrl(s) + "&t=" + Date.now();
}

async function runSmilesPredict() {
  const s = smilesInput.value.trim();
  if (!s) return;
  smilesLoading.value = true;
  smilesResult.value = null;
  try {
    smilesResult.value = await predictFromSmiles(s);
    // Refresh with backend canonical SMILES so the structure preview matches inference input.
    if (smilesResult.value?.smiles) {
      smilesInput.value = smilesResult.value.smiles;
      structurePreviewUrl.value =
        structureSvgUrl(smilesResult.value.smiles) + "&t=" + Date.now();
    }
    ElMessage.success("Prediction complete");
  } catch {
    /* error shown in api */
  } finally {
    smilesLoading.value = false;
  }
}

async function runImagePredict() {
  if (!selectedFile.value) return;
  imageLoading.value = true;
  imageResult.value = null;
  try {
    // Use current SMILES as a reference for same-compound matching on the backend.
    const ref = smilesInput.value.trim() || smilesResult.value?.smiles || "";
    const res = await predictFromImage(selectedFile.value, ref || undefined);
    parsedSmiles.value = res.recognized_smiles || res.smiles;
    imageResult.value = res;
    smilesInput.value = res.smiles;
    structurePreviewUrl.value = structureSvgUrl(res.smiles) + "&t=" + Date.now();
    // Keep the SMILES-side result aligned when the image path resolves to the same structure.
    if (res.matched_reference || res.parsed_from === "image_metadata") {
      smilesResult.value = {
        smiles: res.smiles,
        tg_celsius: res.tg_celsius,
        unit: res.unit || "C",
      };
    }
    let via = "DECIMER recognition";
    if (res.parsed_from === "image_metadata") {
      via = "structure-image metadata";
    } else if (res.matched_reference) {
      via = "matched the input SMILES";
    }
    ElMessage.success(`Recognition and prediction complete (${via})`);
  } catch {
    /* error shown in api */
  } finally {
    imageLoading.value = false;
  }
}

async function repredictFromParsed() {
  const s = parsedSmiles.value.trim();
  if (!s) return;
  smilesLoading.value = true;
  try {
    const res = await predictFromSmiles(s);
    parsedSmiles.value = res.smiles;
    imageResult.value = res;
    smilesInput.value = res.smiles;
    ElMessage.success("Prediction complete");
  } catch {
    /* error shown in api */
  } finally {
    smilesLoading.value = false;
  }
}

watch(activeTab, () => {
  /* keep state when switching tabs */
});
</script>

<style scoped>
.predict-card {
  border-radius: 12px;
  border: 1px solid #d8e2ec;
}
.predict-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}
.panel {
  padding: 1.25rem 0.5rem 0.5rem;
}
.hint {
  color: #5a6d7e;
  font-size: 0.9rem;
  margin: 0 0 1rem;
}
.hint code {
  background: #eef2f6;
  padding: 0.1em 0.35em;
  border-radius: 4px;
}
.actions {
  margin-top: 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.structure-box,
.image-preview {
  margin-top: 1.25rem;
  padding: 1rem;
  background: #fafbfc;
  border: 1px solid #e4eaf0;
  border-radius: 8px;
  text-align: center;
}
.structure-box img,
.image-preview img {
  max-width: 100%;
  max-height: 280px;
}
.upload-icon {
  font-size: 48px;
  color: #4a90e2;
}
.parsed-smiles {
  margin-top: 1rem;
}
</style>
