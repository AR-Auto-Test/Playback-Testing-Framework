package com.google.ar.sceneform.rendering;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

/* loaded from: classes.dex */
public class CleanupItem<T> extends PhantomReference<T> {
    private final Runnable cleanupCallback;

    public CleanupItem(T t, ReferenceQueue<T> referenceQueue, Runnable runnable) {
        super(t, referenceQueue);
        this.cleanupCallback = runnable;
    }

    public void run() {
        this.cleanupCallback.run();
    }
}