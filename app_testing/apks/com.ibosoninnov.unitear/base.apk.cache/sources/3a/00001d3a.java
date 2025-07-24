package com.google.ar.sceneform.resources;

/* loaded from: classes.dex */
public abstract class SharedReference {
    private int referenceCount = 0;

    private void dispose() {
        if (this.referenceCount > 0) {
            return;
        }
        onDispose();
    }

    public abstract void onDispose();

    public void release() {
        this.referenceCount--;
        dispose();
    }

    public void retain() {
        this.referenceCount++;
    }
}