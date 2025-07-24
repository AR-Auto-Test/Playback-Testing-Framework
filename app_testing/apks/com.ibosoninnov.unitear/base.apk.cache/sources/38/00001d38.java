package com.google.ar.sceneform.resources;

/* loaded from: classes.dex */
public interface ResourceHolder {
    void destroyAllResources();

    long reclaimReleasedResources();
}