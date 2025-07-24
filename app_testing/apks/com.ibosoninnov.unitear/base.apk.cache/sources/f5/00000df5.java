package com.google.android.gms.common.api.internal;

import android.os.Bundle;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.common.internal.ClientSettings;
import com.google.android.gms.common.internal.Preconditions;
import java.util.concurrent.locks.Lock;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zaao implements GoogleApiClient.ConnectionCallbacks, GoogleApiClient.OnConnectionFailedListener {
    private final /* synthetic */ zaad zaa;

    private zaao(zaad zaadVar) {
        this.zaa = zaadVar;
    }

    @Override // com.google.android.gms.common.api.internal.ConnectionCallbacks
    public final void onConnected(Bundle bundle) {
        ClientSettings clientSettings;
        com.google.android.gms.signin.zad zadVar;
        clientSettings = this.zaa.zar;
        ClientSettings clientSettings2 = (ClientSettings) Preconditions.checkNotNull(clientSettings);
        zadVar = this.zaa.zak;
        ((com.google.android.gms.signin.zad) Preconditions.checkNotNull(zadVar)).zaa(new zaam(this.zaa));
    }

    @Override // com.google.android.gms.common.api.internal.OnConnectionFailedListener
    public final void onConnectionFailed(ConnectionResult connectionResult) {
        Lock lock;
        Lock lock2;
        boolean zaa;
        lock = this.zaa.zab;
        lock.lock();
        try {
            zaa = this.zaa.zaa(connectionResult);
            if (zaa) {
                this.zaa.zag();
                this.zaa.zae();
            } else {
                this.zaa.zab(connectionResult);
            }
        } finally {
            lock2 = this.zaa.zab;
            lock2.unlock();
        }
    }

    @Override // com.google.android.gms.common.api.internal.ConnectionCallbacks
    public final void onConnectionSuspended(int i) {
    }

    public /* synthetic */ zaao(zaad zaadVar, zaag zaagVar) {
        this(zaadVar);
    }
}