package c.c.a.n;

import android.annotation.SuppressLint;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.util.Log;
import c.c.a.i;
import c.c.a.n.c;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Objects;

/* compiled from: DefaultConnectivityMonitor.java */
/* loaded from: classes.dex */
public final class e implements c {

    /* renamed from: b  reason: collision with root package name */
    public final Context f4078b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a f4079c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f4080d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f4081e;

    /* renamed from: f  reason: collision with root package name */
    public final BroadcastReceiver f4082f = new a();

    /* compiled from: DefaultConnectivityMonitor.java */
    /* loaded from: classes.dex */
    public class a extends BroadcastReceiver {
        public a() {
        }

        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            e eVar = e.this;
            boolean z = eVar.f4080d;
            eVar.f4080d = eVar.i(context);
            if (z != e.this.f4080d) {
                if (Log.isLoggable("ConnectivityMonitor", 3)) {
                    StringBuilder x = c.b.a.a.a.x("connectivity changed, isConnected: ");
                    x.append(e.this.f4080d);
                    Log.d("ConnectivityMonitor", x.toString());
                }
                e eVar2 = e.this;
                c.a aVar = eVar2.f4079c;
                boolean z2 = eVar2.f4080d;
                i.b bVar = (i.b) aVar;
                Objects.requireNonNull(bVar);
                if (z2) {
                    synchronized (c.c.a.i.this) {
                        r rVar = bVar.f3458a;
                        Iterator it = ((ArrayList) c.c.a.s.j.e(rVar.f4098a)).iterator();
                        while (it.hasNext()) {
                            c.c.a.q.c cVar = (c.c.a.q.c) it.next();
                            if (!cVar.i() && !cVar.d()) {
                                cVar.clear();
                                if (!rVar.f4100c) {
                                    cVar.g();
                                } else {
                                    rVar.f4099b.add(cVar);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public e(Context context, c.a aVar) {
        this.f4078b = context.getApplicationContext();
        this.f4079c = aVar;
    }

    @SuppressLint({"MissingPermission"})
    public boolean i(Context context) {
        ConnectivityManager connectivityManager = (ConnectivityManager) context.getSystemService("connectivity");
        Objects.requireNonNull(connectivityManager, "Argument must not be null");
        try {
            NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
            return activeNetworkInfo != null && activeNetworkInfo.isConnected();
        } catch (RuntimeException e2) {
            if (Log.isLoggable("ConnectivityMonitor", 5)) {
                Log.w("ConnectivityMonitor", "Failed to determine connectivity status when connectivity changed", e2);
            }
            return true;
        }
    }

    @Override // c.c.a.n.m
    public void onDestroy() {
    }

    @Override // c.c.a.n.m
    public void onStart() {
        if (this.f4081e) {
            return;
        }
        this.f4080d = i(this.f4078b);
        try {
            this.f4078b.registerReceiver(this.f4082f, new IntentFilter("android.net.conn.CONNECTIVITY_CHANGE"));
            this.f4081e = true;
        } catch (SecurityException e2) {
            if (Log.isLoggable("ConnectivityMonitor", 5)) {
                Log.w("ConnectivityMonitor", "Failed to register", e2);
            }
        }
    }

    @Override // c.c.a.n.m
    public void onStop() {
        if (this.f4081e) {
            this.f4078b.unregisterReceiver(this.f4082f);
            this.f4081e = false;
        }
    }
}