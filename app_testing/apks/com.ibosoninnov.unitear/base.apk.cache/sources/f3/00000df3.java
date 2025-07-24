package com.google.android.gms.common.api.internal;

import java.lang.ref.WeakReference;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zaam extends com.google.android.gms.signin.internal.zad {
    private final WeakReference<zaad> zaa;

    public zaam(zaad zaadVar) {
        this.zaa = new WeakReference<>(zaadVar);
    }

    @Override // com.google.android.gms.signin.internal.zad, com.google.android.gms.signin.internal.zac
    public final void zaa(com.google.android.gms.signin.internal.zam zamVar) {
        zaax zaaxVar;
        zaad zaadVar = this.zaa.get();
        if (zaadVar == null) {
            return;
        }
        zaaxVar = zaadVar.zaa;
        zaaxVar.zaa(new zaal(this, zaadVar, zaadVar, zamVar));
    }
}