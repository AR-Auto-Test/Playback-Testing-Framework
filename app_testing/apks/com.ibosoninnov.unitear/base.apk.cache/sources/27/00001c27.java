package com.google.ar.core;

import java.util.LinkedHashMap;
import java.util.Map;

/* compiled from: FaceCache.java */
/* loaded from: classes.dex */
public final class l extends LinkedHashMap<Long, AugmentedFace> {
    public l() {
        super(1, 0.75f, true);
    }

    @Override // java.util.LinkedHashMap
    public final boolean removeEldestEntry(Map.Entry<Long, AugmentedFace> entry) {
        return size() > 10;
    }
}