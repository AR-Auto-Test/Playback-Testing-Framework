package c.c.a.m;

import java.io.IOException;

/* compiled from: HttpException.java */
/* loaded from: classes.dex */
public final class e extends IOException {
    public e(String str, int i, Throwable th) {
        super(str + ", status code: " + i, th);
    }
}