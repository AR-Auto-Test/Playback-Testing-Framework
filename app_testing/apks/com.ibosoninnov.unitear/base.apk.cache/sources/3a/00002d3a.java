package f.g0.i;

import java.io.IOException;

/* compiled from: StreamResetException.java */
/* loaded from: classes2.dex */
public final class u extends IOException {

    /* renamed from: b  reason: collision with root package name */
    public final b f6006b;

    public u(b bVar) {
        super("stream was reset: " + bVar);
        this.f6006b = bVar;
    }
}