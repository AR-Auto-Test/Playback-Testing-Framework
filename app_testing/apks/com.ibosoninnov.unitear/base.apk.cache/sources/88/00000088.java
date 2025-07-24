package androidx.core.app;

import android.app.PendingIntent;
import androidx.core.graphics.drawable.IconCompat;
import b.b0.a;
import java.util.Objects;

/* loaded from: classes.dex */
public class RemoteActionCompatParcelizer {
    public static RemoteActionCompat read(a aVar) {
        RemoteActionCompat remoteActionCompat = new RemoteActionCompat();
        Object obj = remoteActionCompat.f226a;
        if (aVar.i(1)) {
            obj = aVar.o();
        }
        remoteActionCompat.f226a = (IconCompat) obj;
        CharSequence charSequence = remoteActionCompat.f227b;
        if (aVar.i(2)) {
            charSequence = aVar.h();
        }
        remoteActionCompat.f227b = charSequence;
        CharSequence charSequence2 = remoteActionCompat.f228c;
        if (aVar.i(3)) {
            charSequence2 = aVar.h();
        }
        remoteActionCompat.f228c = charSequence2;
        remoteActionCompat.f229d = (PendingIntent) aVar.m(remoteActionCompat.f229d, 4);
        boolean z = remoteActionCompat.f230e;
        if (aVar.i(5)) {
            z = aVar.f();
        }
        remoteActionCompat.f230e = z;
        boolean z2 = remoteActionCompat.f231f;
        if (aVar.i(6)) {
            z2 = aVar.f();
        }
        remoteActionCompat.f231f = z2;
        return remoteActionCompat;
    }

    public static void write(RemoteActionCompat remoteActionCompat, a aVar) {
        Objects.requireNonNull(aVar);
        IconCompat iconCompat = remoteActionCompat.f226a;
        aVar.p(1);
        aVar.w(iconCompat);
        CharSequence charSequence = remoteActionCompat.f227b;
        aVar.p(2);
        aVar.s(charSequence);
        CharSequence charSequence2 = remoteActionCompat.f228c;
        aVar.p(3);
        aVar.s(charSequence2);
        PendingIntent pendingIntent = remoteActionCompat.f229d;
        aVar.p(4);
        aVar.u(pendingIntent);
        boolean z = remoteActionCompat.f230e;
        aVar.p(5);
        aVar.q(z);
        boolean z2 = remoteActionCompat.f231f;
        aVar.p(6);
        aVar.q(z2);
    }
}