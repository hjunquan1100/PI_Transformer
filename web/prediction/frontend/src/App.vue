<template>
  <div class="app-shell">
    <header class="app-header">
      <h1>Polyimide Tg Prediction and Inverse Design</h1>
      <nav class="app-nav">
        <router-link to="/" class="nav-link" active-class="nav-active">
          Tg Prediction
        </router-link>
        <router-link to="/inverse" class="nav-link" active-class="nav-active">
          Inverse Design
        </router-link>
      </nav>
    </header>
    <main class="app-main">
      <router-view />
    </main>
    <footer class="app-footer">
      <span v-if="healthText">{{ healthText }}</span>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { fetchHealth } from "@/api/predict";

const healthText = ref("");

onMounted(async () => {
  try {
    const h = await fetchHealth();
    const parts = [`Forward: ${h.model_loaded ? "loaded" : "not loaded"}`];
    parts.push(`Inverse: ${h.inverse_model_loaded ? "ready" : "not ready"}`);
    parts.push(`DECIMER: ${h.decimer_available ? "available" : "not installed"}`);
    parts.push(`Device: ${h.device}`);
    healthText.value = parts.join("  /  ");
  } catch {
    healthText.value = "Backend is not connected. Start the API service first.";
  }
});
</script>

<style>
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: linear-gradient(160deg, #f0f4f8 0%, #e8eef5 50%, #dfe8f2 100%);
  min-height: 100vh;
  color: #1a2a3a;
}
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  text-align: center;
  padding: 1.5rem 1rem 0.75rem;
}
.app-header h1 {
  margin: 0 0 1rem;
  font-size: 1.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.app-nav {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.nav-link {
  padding: 0.45rem 1.25rem;
  border-radius: 8px;
  text-decoration: none;
  color: #4a6fa5;
  font-weight: 500;
  border: 1px solid transparent;
  transition: background 0.15s, color 0.15s;
}
.nav-link:hover {
  background: rgba(74, 144, 226, 0.08);
}
.nav-active {
  background: #4a90e2;
  color: #fff;
  border-color: #3a7bc8;
}
.app-main {
  flex: 1;
  max-width: 960px;
  width: 100%;
  margin: 0 auto;
  padding: 0 1.25rem 2rem;
}
.app-footer {
  text-align: center;
  padding: 1rem;
  font-size: 0.8rem;
  color: #7a8d9e;
}
</style>
