package com.google.firebase.platforminfo;

import e.a;

/* loaded from: classes2.dex */
public final class KotlinDetector {
    private KotlinDetector() {
    }

    public static String detectVersion() {
        try {
            return a.f5710b.toString();
        } catch (NoClassDefFoundError unused) {
            return null;
        }
    }
}