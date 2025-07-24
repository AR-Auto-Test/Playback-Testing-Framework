package c.d.d.a;

import com.google.mediapipe.components.ExternalTextureConverter;
import java.lang.Thread;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class g implements Thread.UncaughtExceptionHandler {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ExternalTextureConverter f4483a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Object f4484b;

    public /* synthetic */ g(ExternalTextureConverter externalTextureConverter, Object obj) {
        this.f4483a = externalTextureConverter;
        this.f4484b = obj;
    }

    @Override // java.lang.Thread.UncaughtExceptionHandler
    public final void uncaughtException(Thread thread, Throwable th) {
        this.f4483a.a(this.f4484b, thread, th);
    }
}