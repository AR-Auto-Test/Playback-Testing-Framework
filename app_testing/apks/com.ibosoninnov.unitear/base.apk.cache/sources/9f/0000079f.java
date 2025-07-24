package c.c.a.m.v;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;

/* compiled from: ResourceRecycler.java */
/* loaded from: classes.dex */
public class z {

    /* renamed from: a  reason: collision with root package name */
    public boolean f3819a;

    /* renamed from: b  reason: collision with root package name */
    public final Handler f3820b = new Handler(Looper.getMainLooper(), new a());

    /* compiled from: ResourceRecycler.java */
    /* loaded from: classes.dex */
    public static final class a implements Handler.Callback {
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1) {
                ((w) message.obj).a();
                return true;
            }
            return false;
        }
    }

    public synchronized void a(w<?> wVar, boolean z) {
        if (!this.f3819a && !z) {
            this.f3819a = true;
            wVar.a();
            this.f3819a = false;
        }
        this.f3820b.obtainMessage(1, wVar).sendToTarget();
    }
}