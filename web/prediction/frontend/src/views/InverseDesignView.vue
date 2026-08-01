<template>
  <el-card class="inverse-card" shadow="hover">
    <div class="panel" v-loading="loading" element-loading-text="Generating 15 candidate structures; this may take 1-3 minutes...">
      <p class="hint">
        Enter a target glass transition temperature (deg C). The system generates
        15 candidate polyimide structures and recommends the Top
        <strong>5</strong> structures closest to the target Tg.
      </p>

      <div class="input-row">
        <span class="label">Target Tg</span>
        <el-input-number
          v-model="tgTarget"
          :precision="1"
          :step="0.1"
          :min="19"
          :max="460"
          controls-position="right"
          size="large"
        />
        <span class="unit"> deg C</span>
        <el-button
          type="primary"
          size="large"
          :loading="loading"
          @click="runGenerate"
        >
          Generate Structures
        </el-button>
      </div>

      <el-alert
        v-if="result"
        type="info"
        :closable="false"
        class="summary-alert"
        show-icon
      >
        Target {{ result.tg_target_c }} deg C /
        generated {{ result.n_generated }} /
        RDKit valid {{ result.valid_count }} /
        PI passed {{ result.passed_count }} /
        recommended {{ result.recommended.length }}
      </el-alert>

      <section v-if="result?.recommended.length" class="section">
        <h2 class="section-title">Recommended Structures (Top 5)</h2>
        <div
          v-for="mol in result.recommended"
          :key="mol.id"
          class="mol-card"
        >
          <div class="mol-header">
            <span class="rank">#{{ mol.rank }}</span>
            <span class="pred">
              Predicted Tg <strong>{{ mol.pred_tg_c.toFixed(1) }}</strong> deg C
              <span class="err"> (error {{ formatError(mol.tg_error_c) }}  deg C)</span>
            </span>
            <el-button size="small" text type="primary" @click="copySmiles(mol.smiles)">
              Copy SMILES
            </el-button>
          </div>
          <code class="smiles">{{ mol.smiles }}</code>
          <div class="structure-box">
            <img
              :src="structureSvgUrl(mol.smiles) + '&t=' + cacheBust"
              :alt="'structure ' + mol.rank"
            />
          </div>
        </div>
      </section>

      <el-collapse v-if="result?.others.length" class="others-collapse">
        <el-collapse-item title="Other validated structures" :name="1">
          <div
            v-for="mol in result.others"
            :key="mol.id"
            class="mol-card mol-card-compact"
          >
            <div class="mol-header">
              <span class="rank">#{{ mol.rank }}</span>
              <span class="pred">
                Predicted Tg {{ mol.pred_tg_c.toFixed(1) }} deg C (error {{ formatError(mol.tg_error_c) }} deg C)
              </span>
            </div>
            <code class="smiles">{{ mol.smiles }}</code>
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import { structureSvgUrl } from "@/api/client";
import {
  generateInverse,
  type InverseGenerateResponse,
} from "@/api/inverse";

const tgTarget = ref(300.0);
const loading = ref(false);
const result = ref<InverseGenerateResponse | null>(null);
const cacheBust = ref(Date.now());

function formatError(err: number): string {
  return err >= 0 ? `+${err.toFixed(1)}` : err.toFixed(1);
}

async function runGenerate() {
  loading.value = true;
  result.value = null;
  try {
    const tg = Math.round(tgTarget.value * 10) / 10;
    tgTarget.value = tg;
    result.value = await generateInverse(tg);
    cacheBust.value = Date.now();
    ElMessage.success(`Generated ${result.value.recommended.length} recommended structures`);
  } catch {
    /* error shown in api */
  } finally {
    loading.value = false;
  }
}

async function copySmiles(smiles: string) {
  try {
    await navigator.clipboard.writeText(smiles);
    ElMessage.success("SMILES copied");
  } catch {
    ElMessage.warning("Copy failed; select the text manually");
  }
}
</script>

<style scoped>
.inverse-card {
  border-radius: 12px;
  border: 1px solid #d8e2ec;
}
.panel {
  padding: 1.25rem 0.5rem 0.5rem;
}
.hint {
  color: #5a6d7e;
  font-size: 0.9rem;
  margin: 0 0 1.25rem;
  line-height: 1.6;
}
.input-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
}
.label {
  font-weight: 500;
  color: #3d5166;
}
.unit {
  color: #5a6d7e;
}
.summary-alert {
  margin-bottom: 1.25rem;
}
.section-title {
  font-size: 1.1rem;
  margin: 0 0 1rem;
  color: #2c5282;
}
.mol-card {
  border: 1px solid #e4eaf0;
  border-radius: 10px;
  padding: 1rem;
  margin-bottom: 1rem;
  background: #fafbfc;
}
.mol-card-compact {
  padding: 0.75rem;
}
.mol-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 1rem;
  margin-bottom: 0.5rem;
}
.rank {
  font-weight: 700;
  color: #2b6cb0;
}
.pred .err {
  color: #718096;
  font-size: 0.9em;
}
.smiles {
  display: block;
  word-break: break-all;
  font-size: 0.82rem;
  line-height: 1.5;
  margin-bottom: 0.75rem;
  color: #2d3748;
}
.structure-box {
  text-align: center;
  padding: 0.5rem;
  background: #fff;
  border-radius: 8px;
  border: 1px solid #edf2f7;
}
.structure-box img {
  max-width: 100%;
  max-height: 260px;
}
.others-collapse {
  margin-top: 0.5rem;
}
</style>
