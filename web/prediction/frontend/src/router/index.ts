import { createRouter, createWebHistory } from "vue-router";
import TgPredictView from "@/views/TgPredictView.vue";
import InverseDesignView from "@/views/InverseDesignView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "predict",
      component: TgPredictView,
      meta: { title: "Tg prediction" },
    },
    {
      path: "/inverse",
      name: "inverse",
      component: InverseDesignView,
      meta: { title: "inverse design" },
    },
  ],
});

export default router;
