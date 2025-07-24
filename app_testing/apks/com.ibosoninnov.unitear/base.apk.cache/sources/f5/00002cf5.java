package f.g0;

import java.util.concurrent.ThreadFactory;

/* compiled from: Util.java */
/* loaded from: classes2.dex */
public final class d implements ThreadFactory {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f5781a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ boolean f5782b;

    public d(String str, boolean z) {
        this.f5781a = str;
        this.f5782b = z;
    }

    @Override // java.util.concurrent.ThreadFactory
    public Thread newThread(Runnable runnable) {
        Thread thread = new Thread(runnable, this.f5781a);
        thread.setDaemon(this.f5782b);
        return thread;
    }
}