package pl.droidsonroids.gif;

import h.a.a.d;
import java.io.IOException;

/* loaded from: classes2.dex */
public class GifIOException extends IOException {

    /* renamed from: b  reason: collision with root package name */
    public final d f6264b;

    /* renamed from: c  reason: collision with root package name */
    public final String f6265c;

    public GifIOException(int i, String str) {
        d dVar;
        d[] values = d.values();
        int i2 = 0;
        while (true) {
            if (i2 < 21) {
                dVar = values[i2];
                if (dVar.y == i) {
                    break;
                }
                i2++;
            } else {
                dVar = d.UNKNOWN;
                dVar.y = i;
                break;
            }
        }
        this.f6264b = dVar;
        this.f6265c = str;
    }

    @Override // java.lang.Throwable
    public String getMessage() {
        if (this.f6265c == null) {
            return this.f6264b.a();
        }
        return this.f6264b.a() + ": " + this.f6265c;
    }
}