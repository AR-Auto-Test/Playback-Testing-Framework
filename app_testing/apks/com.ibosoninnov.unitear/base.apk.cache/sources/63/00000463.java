package b.j.g;

import android.os.Process;
import java.util.concurrent.ThreadFactory;

/* compiled from: RequestExecutor.java */
/* loaded from: classes.dex */
public class n implements ThreadFactory {

    /* renamed from: a  reason: collision with root package name */
    public String f2157a;

    /* renamed from: b  reason: collision with root package name */
    public int f2158b;

    /* compiled from: RequestExecutor.java */
    /* loaded from: classes.dex */
    public static class a extends Thread {

        /* renamed from: b  reason: collision with root package name */
        public final int f2159b;

        public a(Runnable runnable, String str, int i) {
            super(runnable, str);
            this.f2159b = i;
        }

        @Override // java.lang.Thread, java.lang.Runnable
        public void run() {
            Process.setThreadPriority(this.f2159b);
            super.run();
        }
    }

    public n(String str, int i) {
        this.f2157a = str;
        this.f2158b = i;
    }

    @Override // java.util.concurrent.ThreadFactory
    public Thread newThread(Runnable runnable) {
        return new a(runnable, this.f2157a, this.f2158b);
    }
}