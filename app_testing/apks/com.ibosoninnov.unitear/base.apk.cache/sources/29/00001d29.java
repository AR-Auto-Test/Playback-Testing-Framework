package com.google.ar.sceneform.rendering;

import android.os.AsyncTask;
import android.os.Handler;
import android.os.Looper;
import java.util.concurrent.Executor;

/* loaded from: classes.dex */
public class ThreadPools {
    private static Executor mainExecutor;
    private static Executor threadPoolExecutor;

    private ThreadPools() {
    }

    public static Executor getMainExecutor() {
        if (mainExecutor == null) {
            mainExecutor = new Executor() { // from class: com.google.ar.sceneform.rendering.ThreadPools.1
                private final Handler handler = new Handler(Looper.getMainLooper());

                @Override // java.util.concurrent.Executor
                public void execute(Runnable runnable) {
                    this.handler.post(runnable);
                }
            };
        }
        return mainExecutor;
    }

    public static Executor getThreadPoolExecutor() {
        Executor executor = threadPoolExecutor;
        return executor == null ? AsyncTask.THREAD_POOL_EXECUTOR : executor;
    }

    public static void setMainExecutor(Executor executor) {
        mainExecutor = executor;
    }

    public static void setThreadPoolExecutor(Executor executor) {
        threadPoolExecutor = executor;
    }
}