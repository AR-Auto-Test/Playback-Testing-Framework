package h.a.a;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import java.lang.ref.WeakReference;
import java.util.Iterator;

/* compiled from: InvalidationHandler.java */
/* loaded from: classes2.dex */
public class h extends Handler {

    /* renamed from: a  reason: collision with root package name */
    public final WeakReference<c> f6250a;

    public h(c cVar) {
        super(Looper.getMainLooper());
        this.f6250a = new WeakReference<>(cVar);
    }

    @Override // android.os.Handler
    public void handleMessage(Message message) {
        c cVar = this.f6250a.get();
        if (cVar == null) {
            return;
        }
        if (message.what == -1) {
            cVar.invalidateSelf();
            return;
        }
        Iterator<a> it = cVar.i.iterator();
        while (it.hasNext()) {
            it.next().a(message.what);
        }
    }
}