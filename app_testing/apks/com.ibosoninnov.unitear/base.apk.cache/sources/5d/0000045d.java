package b.j.g;

import b.j.g.j;
import java.util.ArrayList;

/* compiled from: FontRequestWorker.java */
/* loaded from: classes.dex */
public class i implements b.j.i.a<j.a> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f2143a;

    public i(String str) {
        this.f2143a = str;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // b.j.i.a
    public void accept(j.a aVar) {
        j.a aVar2 = aVar;
        synchronized (j.f2146c) {
            b.f.h<String, ArrayList<b.j.i.a<j.a>>> hVar = j.f2147d;
            ArrayList<b.j.i.a<j.a>> arrayList = hVar.get(this.f2143a);
            if (arrayList == null) {
                return;
            }
            hVar.remove(this.f2143a);
            for (int i = 0; i < arrayList.size(); i++) {
                arrayList.get(i).accept(aVar2);
            }
        }
    }
}