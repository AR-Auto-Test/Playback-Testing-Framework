package com.google.ar.core;

import java.util.Map;

/* compiled from: FaceCache.java */
/* loaded from: classes.dex */
public final class m {

    /* renamed from: a  reason: collision with root package name */
    public final Map<Long, AugmentedFace> f5589a = new l();

    public final synchronized AugmentedFace a(long j, Session session) {
        Map<Long, AugmentedFace> map = this.f5589a;
        Long valueOf = Long.valueOf(j);
        AugmentedFace augmentedFace = map.get(valueOf);
        if (augmentedFace == null) {
            AugmentedFace augmentedFace2 = new AugmentedFace(j, session);
            this.f5589a.put(valueOf, augmentedFace2);
            return augmentedFace2;
        }
        return augmentedFace;
    }
}