package com.google.ar.sceneform;

import android.annotation.TargetApi;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

@TargetApi(24)
/* loaded from: classes.dex */
public class SequentialTask {
    private CompletableFuture<Void> future;

    public CompletableFuture<Void> appendRunnable(Runnable runnable, Executor executor) {
        CompletableFuture<Void> completableFuture = this.future;
        if (completableFuture != null && !completableFuture.isDone()) {
            this.future = this.future.thenRunAsync(runnable, executor);
        } else {
            this.future = CompletableFuture.runAsync(runnable, executor);
        }
        return this.future;
    }

    public boolean isDone() {
        CompletableFuture<Void> completableFuture = this.future;
        if (completableFuture == null) {
            return true;
        }
        if (completableFuture.isDone()) {
            this.future = null;
            return true;
        }
        return false;
    }
}