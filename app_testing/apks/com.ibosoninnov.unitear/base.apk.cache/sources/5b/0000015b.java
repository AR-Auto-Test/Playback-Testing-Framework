package b.b.c;

import android.content.Context;
import android.location.Location;
import android.location.LocationManager;
import android.util.Log;

/* compiled from: TwilightManager.java */
/* loaded from: classes.dex */
public class t {

    /* renamed from: a  reason: collision with root package name */
    public static t f608a;

    /* renamed from: b  reason: collision with root package name */
    public final Context f609b;

    /* renamed from: c  reason: collision with root package name */
    public final LocationManager f610c;

    /* renamed from: d  reason: collision with root package name */
    public final a f611d = new a();

    /* compiled from: TwilightManager.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public boolean f612a;

        /* renamed from: b  reason: collision with root package name */
        public long f613b;
    }

    public t(Context context, LocationManager locationManager) {
        this.f609b = context;
        this.f610c = locationManager;
    }

    public final Location a(String str) {
        try {
            if (this.f610c.isProviderEnabled(str)) {
                return this.f610c.getLastKnownLocation(str);
            }
            return null;
        } catch (Exception e2) {
            Log.d("TwilightManager", "Failed to get last known location", e2);
            return null;
        }
    }
}