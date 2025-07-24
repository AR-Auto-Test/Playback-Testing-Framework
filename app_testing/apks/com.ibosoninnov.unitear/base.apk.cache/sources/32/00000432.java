package b.j.c;

import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.os.Process;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

/* compiled from: ContextCompat.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: a  reason: collision with root package name */
    public static final Object f2074a = new Object();

    /* compiled from: ContextCompat.java */
    /* renamed from: b.j.c.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class ExecutorC0033a implements Executor {

        /* renamed from: b  reason: collision with root package name */
        public final Handler f2075b;

        public ExecutorC0033a(Handler handler) {
            this.f2075b = handler;
        }

        @Override // java.util.concurrent.Executor
        public void execute(Runnable runnable) {
            if (this.f2075b.post(runnable)) {
                return;
            }
            throw new RejectedExecutionException(this.f2075b + " is shutting down");
        }
    }

    public static int a(Context context, String str) {
        if (str != null) {
            return context.checkPermission(str, Process.myPid(), Process.myUid());
        }
        throw new IllegalArgumentException("permission is null");
    }

    public static Executor b(Context context) {
        if (Build.VERSION.SDK_INT >= 28) {
            return context.getMainExecutor();
        }
        return new ExecutorC0033a(new Handler(context.getMainLooper()));
    }
}