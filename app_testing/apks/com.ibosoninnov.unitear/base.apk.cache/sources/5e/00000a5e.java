package c.e.b.p000if;

import android.content.Context;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.util.Log;
import c.e.b.gf.c;
import c.e.b.ve;
import java.io.IOException;
import java.net.URL;

/* compiled from: MyAyncTask.java */
/* renamed from: c.e.b.if.m  reason: invalid package */
/* loaded from: classes2.dex */
public class m extends AsyncTask<ve, Integer, String> {

    /* renamed from: a  reason: collision with root package name */
    public c f4892a;

    /* renamed from: b  reason: collision with root package name */
    public e f4893b;

    public m(Context context, c cVar) {
        this.f4892a = cVar;
        this.f4893b = new e(context);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // android.os.AsyncTask
    public String doInBackground(ve[] veVarArr) {
        Log.i("ffffffffffffffffffffffffffffffffffff", "start");
        for (String str : veVarArr[0].T.values()) {
            String substring = str.substring(str.lastIndexOf(47) + 1);
            try {
                this.f4893b.b(BitmapFactory.decodeStream(new URL(str).openConnection().getInputStream()), substring);
            } catch (IOException e2) {
                System.out.println(e2);
            }
        }
        return "null";
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // android.os.AsyncTask
    public void onPostExecute(String str) {
        this.f4892a.b("id", "filePath");
        Log.i("ffffffffffffffffffffffffffffffffffff", "end");
    }

    @Override // android.os.AsyncTask
    public void onPreExecute() {
        super.onPreExecute();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    @Override // android.os.AsyncTask
    public void onProgressUpdate(Integer[] numArr) {
        super.onProgressUpdate(numArr);
    }
}