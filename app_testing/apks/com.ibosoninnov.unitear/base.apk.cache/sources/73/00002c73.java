package com.google.vr.dynamite.client;

/* compiled from: TargetLibraryInfo.java */
/* loaded from: classes2.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    private final String f5655a;

    /* renamed from: b  reason: collision with root package name */
    private final String f5656b;

    public g(String str, String str2) {
        this.f5655a = str;
        this.f5656b = str2;
    }

    public final String a() {
        return this.f5655a;
    }

    public final boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof g) {
            g gVar = (g) obj;
            if (f.a(this.f5655a, gVar.f5655a) && f.a(this.f5656b, gVar.f5656b)) {
                return true;
            }
        }
        return false;
    }

    public final int hashCode() {
        return f.b(this.f5656b) + (f.b(this.f5655a) * 37);
    }

    public final String toString() {
        StringBuilder sb = new StringBuilder("[packageName=");
        sb.append(this.f5655a);
        sb.append(",libraryName=");
        return c.b.a.a.a.v(sb, this.f5656b, "]");
    }
}