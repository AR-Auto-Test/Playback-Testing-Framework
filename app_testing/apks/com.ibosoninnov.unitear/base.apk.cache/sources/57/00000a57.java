package c.e.b.p000if;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.util.Log;
import java.io.IOException;
import java.util.Calendar;

/* compiled from: CheckNetworkConnection.java */
/* renamed from: c.e.b.if.g  reason: invalid package */
/* loaded from: classes2.dex */
public class g extends AsyncTask<Void, Void, Boolean> {

    /* renamed from: a  reason: collision with root package name */
    public final a f4875a;

    /* renamed from: b  reason: collision with root package name */
    public final Context f4876b;

    /* compiled from: CheckNetworkConnection.java */
    /* renamed from: c.e.b.if.g$a */
    /* loaded from: classes2.dex */
    public interface a {
        void a();

        void b(String str);
    }

    public g(Context context, a aVar) {
        this.f4875a = aVar;
        this.f4876b = context;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    /* JADX WARN: Code restructure failed: missing block: B:33:0x00a2, code lost:
        if (r5 == false) goto L16;
     */
    @Override // android.os.AsyncTask
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public Boolean doInBackground(Void[] voidArr) {
        boolean z;
        boolean z2;
        long timeInMillis;
        StringBuilder sb;
        boolean z3;
        Context context = this.f4876b;
        if (context == null) {
            return Boolean.FALSE;
        }
        NetworkInfo activeNetworkInfo = ((ConnectivityManager) context.getSystemService("connectivity")).getActiveNetworkInfo();
        boolean z4 = true;
        if (activeNetworkInfo != null) {
            z2 = activeNetworkInfo.getType() == 1;
            z = activeNetworkInfo.getType() == 0;
        } else {
            z = false;
            z2 = false;
        }
        if (z2 || z) {
            long timeInMillis2 = Calendar.getInstance().getTimeInMillis();
            try {
                try {
                    z3 = Runtime.getRuntime().exec("/system/bin/ping -c 1 8.8.8.8").waitFor() == 0;
                } catch (IOException e2) {
                    e2.printStackTrace();
                    timeInMillis = Calendar.getInstance().getTimeInMillis();
                    sb = new StringBuilder();
                    sb.append(timeInMillis - timeInMillis2);
                    sb.append("");
                    Log.i("NetWork check Time", sb.toString());
                    z3 = false;
                } catch (InterruptedException e3) {
                    e3.printStackTrace();
                    timeInMillis = Calendar.getInstance().getTimeInMillis();
                    sb = new StringBuilder();
                    sb.append(timeInMillis - timeInMillis2);
                    sb.append("");
                    Log.i("NetWork check Time", sb.toString());
                    z3 = false;
                }
            } finally {
                long timeInMillis3 = Calendar.getInstance().getTimeInMillis();
                Log.i("NetWork check Time", (timeInMillis3 - timeInMillis2) + "");
            }
        }
        z4 = false;
        return Boolean.valueOf(z4);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // android.os.AsyncTask
    public void onPostExecute(Boolean bool) {
        Boolean bool2 = bool;
        super.onPostExecute(bool2);
        if (bool2.booleanValue()) {
            this.f4875a.a();
        } else {
            this.f4875a.b(this.f4876b == null ? "Context is null" : "No Internet Connection");
        }
    }

    @Override // android.os.AsyncTask
    public void onPreExecute() {
        super.onPreExecute();
    }
}