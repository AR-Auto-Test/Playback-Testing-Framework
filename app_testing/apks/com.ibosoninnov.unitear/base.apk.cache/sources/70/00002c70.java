package com.google.vr.dynamite.client;

/* compiled from: LoaderException.java */
/* loaded from: classes2.dex */
public final class d extends Exception {

    /* renamed from: a  reason: collision with root package name */
    private final int f5651a = 1;

    @Override // java.lang.Throwable
    public final String getMessage() {
        String str = this.f5651a != 1 ? "Unknown error" : "Package not available";
        StringBuilder sb = new StringBuilder(str.length() + 17);
        sb.append("LoaderException{");
        sb.append(str);
        sb.append("}");
        return sb.toString();
    }
}