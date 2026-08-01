<template>
  <el-card class="result-panel" shadow="never">
    <template #header>
      <span class="result-title">Prediction Result</span>
    </template>
    <div class="tg-value">
      <span class="label">Glass transition temperature Tg</span>
      <span class="value">{{ result.tg_celsius.toFixed(1) }}</span>
      <span class="unit"> deg C</span>
    </div>
    <div v-if="structureUrl" class="structure-box">
      <img :src="structureUrl" alt="2D structure" />
    </div>
    <el-descriptions :column="1" border size="small" class="meta">
      <el-descriptions-item label="SMILES">
        <code class="smiles-code">{{ result.smiles }}</code>
      </el-descriptions-item>
      <el-descriptions-item v-if="result.parsed_from" label="Source">
        Structure image recognition (DECIMER)
      </el-descriptions-item>
    </el-descriptions>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PredictResponse } from "@/api/predict";
import { structureSvgUrl } from "@/api/predict";

const props = defineProps<{
  result: PredictResponse;
}>();

const structureUrl = computed(() =>
  props.result.smiles?.trim()
    ? structureSvgUrl(props.result.smiles.trim()) + "&t=" + encodeURIComponent(props.result.smiles)
    : ""
);
</script>

<style scoped>
.result-panel {
  margin-top: 1.5rem;
  background: linear-gradient(135deg, #f8fbff 0%, #f0f6fc 100%);
  border-color: #c5d9ed;
}
.result-title {
  font-weight: 600;
  color: #2c5282;
}
.tg-value {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.tg-value .label {
  color: #5a6d7e;
  font-size: 0.95rem;
}
.tg-value .value {
  font-size: 2.25rem;
  font-weight: 700;
  color: #2b6cb0;
}
.tg-value .unit {
  font-size: 1.1rem;
  color: #4a6fa5;
}
.structure-box {
  margin: 0 0 1rem;
  padding: 0.75rem;
  background: #fff;
  border: 1px solid #e4eaf0;
  border-radius: 8px;
  text-align: center;
}
.structure-box img {
  max-width: 100%;
  max-height: 280px;
}
.smiles-code {
  word-break: break-all;
  font-size: 0.85rem;
  line-height: 1.5;
}
</style>
