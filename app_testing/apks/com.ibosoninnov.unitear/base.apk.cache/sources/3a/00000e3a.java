package com.google.android.gms.common.api.internal;

import android.app.PendingIntent;
import android.content.Context;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import b.f.a;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailabilityLight;
import com.google.android.gms.common.api.Api;
import com.google.android.gms.common.api.Result;
import com.google.android.gms.common.api.Status;
import com.google.android.gms.common.api.internal.BaseImplementation;
import com.google.android.gms.common.internal.ClientSettings;
import com.google.android.gms.common.internal.Objects;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.signin.SignInOptions;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zaq implements zabn {
    private final Context zaa;
    private final zaap zab;
    private final Looper zac;
    private final zaax zad;
    private final zaax zae;
    private final Map<Api.AnyClientKey<?>, zaax> zaf;
    private final Api.Client zah;
    private Bundle zai;
    private final Lock zam;
    private final Set<SignInConnectionListener> zag = Collections.newSetFromMap(new WeakHashMap());
    private ConnectionResult zaj = null;
    private ConnectionResult zak = null;
    private boolean zal = false;
    private int zan = 0;

    private zaq(Context context, zaap zaapVar, Lock lock, Looper looper, GoogleApiAvailabilityLight googleApiAvailabilityLight, Map<Api.AnyClientKey<?>, Api.Client> map, Map<Api.AnyClientKey<?>, Api.Client> map2, ClientSettings clientSettings, Api.AbstractClientBuilder<? extends com.google.android.gms.signin.zad, SignInOptions> abstractClientBuilder, Api.Client client, ArrayList<zap> arrayList, ArrayList<zap> arrayList2, Map<Api<?>, Boolean> map3, Map<Api<?>, Boolean> map4) {
        this.zaa = context;
        this.zab = zaapVar;
        this.zam = lock;
        this.zac = looper;
        this.zah = client;
        this.zad = new zaax(context, zaapVar, lock, looper, googleApiAvailabilityLight, map2, null, map4, null, arrayList2, new zas(this, null));
        this.zae = new zaax(context, zaapVar, lock, looper, googleApiAvailabilityLight, map, clientSettings, map3, abstractClientBuilder, arrayList, new zau(this, null));
        a aVar = new a();
        for (Api.AnyClientKey<?> anyClientKey : map2.keySet()) {
            aVar.put(anyClientKey, this.zad);
        }
        for (Api.AnyClientKey<?> anyClientKey2 : map.keySet()) {
            aVar.put(anyClientKey2, this.zae);
        }
        this.zaf = Collections.unmodifiableMap(aVar);
    }

    public static zaq zaa(Context context, zaap zaapVar, Lock lock, Looper looper, GoogleApiAvailabilityLight googleApiAvailabilityLight, Map<Api.AnyClientKey<?>, Api.Client> map, ClientSettings clientSettings, Map<Api<?>, Boolean> map2, Api.AbstractClientBuilder<? extends com.google.android.gms.signin.zad, SignInOptions> abstractClientBuilder, ArrayList<zap> arrayList) {
        a aVar = new a();
        a aVar2 = new a();
        Api.Client client = null;
        for (Map.Entry<Api.AnyClientKey<?>, Api.Client> entry : map.entrySet()) {
            Api.Client value = entry.getValue();
            if (value.providesSignIn()) {
                client = value;
            }
            if (value.requiresSignIn()) {
                aVar.put(entry.getKey(), value);
            } else {
                aVar2.put(entry.getKey(), value);
            }
        }
        Preconditions.checkState(!aVar.isEmpty(), "CompositeGoogleApiClient should not be used without any APIs that require sign-in.");
        a aVar3 = new a();
        a aVar4 = new a();
        for (Api<?> api : map2.keySet()) {
            Api.AnyClientKey<?> zac = api.zac();
            if (aVar.containsKey(zac)) {
                aVar3.put(api, map2.get(api));
            } else if (aVar2.containsKey(zac)) {
                aVar4.put(api, map2.get(api));
            } else {
                throw new IllegalStateException("Each API in the isOptionalMap must have a corresponding client in the clients map.");
            }
        }
        ArrayList arrayList2 = new ArrayList();
        ArrayList arrayList3 = new ArrayList();
        int size = arrayList.size();
        int i = 0;
        while (i < size) {
            zap zapVar = arrayList.get(i);
            i++;
            zap zapVar2 = zapVar;
            if (aVar3.containsKey(zapVar2.zaa)) {
                arrayList2.add(zapVar2);
            } else if (aVar4.containsKey(zapVar2.zaa)) {
                arrayList3.add(zapVar2);
            } else {
                throw new IllegalStateException("Each ClientCallbacks must have a corresponding API in the isOptionalMap");
            }
        }
        return new zaq(context, zaapVar, lock, looper, googleApiAvailabilityLight, aVar, aVar2, clientSettings, abstractClientBuilder, client, arrayList2, arrayList3, aVar3, aVar4);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zah() {
        ConnectionResult connectionResult;
        if (zab(this.zaj)) {
            if (!zab(this.zak) && !zaj()) {
                ConnectionResult connectionResult2 = this.zak;
                if (connectionResult2 != null) {
                    if (this.zan == 1) {
                        zai();
                        return;
                    }
                    zaa(connectionResult2);
                    this.zad.zac();
                    return;
                }
                return;
            }
            int i = this.zan;
            if (i != 1) {
                if (i != 2) {
                    Log.wtf("CompositeGAC", "Attempted to call success callbacks in CONNECTION_MODE_NONE. Callbacks should be disabled via GmsClientSupervisor", new AssertionError());
                    this.zan = 0;
                }
                ((zaap) Preconditions.checkNotNull(this.zab)).zaa(this.zai);
            }
            zai();
            this.zan = 0;
        } else if (this.zaj != null && zab(this.zak)) {
            this.zae.zac();
            zaa((ConnectionResult) Preconditions.checkNotNull(this.zaj));
        } else {
            ConnectionResult connectionResult3 = this.zaj;
            if (connectionResult3 == null || (connectionResult = this.zak) == null) {
                return;
            }
            if (this.zae.zac < this.zad.zac) {
                connectionResult3 = connectionResult;
            }
            zaa(connectionResult3);
        }
    }

    private final void zai() {
        for (SignInConnectionListener signInConnectionListener : this.zag) {
            signInConnectionListener.onComplete();
        }
        this.zag.clear();
    }

    private final boolean zaj() {
        ConnectionResult connectionResult = this.zak;
        return connectionResult != null && connectionResult.getErrorCode() == 4;
    }

    private final PendingIntent zak() {
        if (this.zah == null) {
            return null;
        }
        return PendingIntent.getActivity(this.zaa, System.identityHashCode(this.zab), this.zah.getSignInIntent(), 134217728);
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final <A extends Api.AnyClient, T extends BaseImplementation.ApiMethodImpl<? extends Result, A>> T zab(T t) {
        if (zac((BaseImplementation.ApiMethodImpl<? extends Result, ? extends Api.AnyClient>) t)) {
            if (zaj()) {
                t.setFailedResult(new Status(4, (String) null, zak()));
                return t;
            }
            return (T) this.zae.zab((zaax) t);
        }
        return (T) this.zad.zab((zaax) t);
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final void zac() {
        this.zak = null;
        this.zaj = null;
        this.zan = 0;
        this.zad.zac();
        this.zae.zac();
        zai();
    }

    /* JADX WARN: Code restructure failed: missing block: B:10:0x001e, code lost:
        if (r2.zan == 1) goto L12;
     */
    @Override // com.google.android.gms.common.api.internal.zabn
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean zad() {
        this.zam.lock();
        try {
            boolean z = true;
            if (this.zad.zad()) {
                if (!this.zae.zad() && !zaj()) {
                }
                return z;
            }
            z = false;
            return z;
        } finally {
            this.zam.unlock();
        }
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final boolean zae() {
        this.zam.lock();
        try {
            return this.zan == 2;
        } finally {
            this.zam.unlock();
        }
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final void zaf() {
        this.zad.zaf();
        this.zae.zaf();
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final void zag() {
        this.zam.lock();
        try {
            boolean zae = zae();
            this.zae.zac();
            this.zak = new ConnectionResult(4);
            if (zae) {
                new com.google.android.gms.internal.base.zap(this.zac).post(new zat(this));
            } else {
                zai();
            }
        } finally {
            this.zam.unlock();
        }
    }

    private final boolean zac(BaseImplementation.ApiMethodImpl<? extends Result, ? extends Api.AnyClient> apiMethodImpl) {
        zaax zaaxVar = this.zaf.get(apiMethodImpl.getClientKey());
        Preconditions.checkNotNull(zaaxVar, "GoogleApiClient is not configured to use the API required for this call.");
        return zaaxVar.equals(this.zae);
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final ConnectionResult zab() {
        throw new UnsupportedOperationException();
    }

    private static boolean zab(ConnectionResult connectionResult) {
        return connectionResult != null && connectionResult.isSuccess();
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final <A extends Api.AnyClient, R extends Result, T extends BaseImplementation.ApiMethodImpl<R, A>> T zaa(T t) {
        if (zac((BaseImplementation.ApiMethodImpl<? extends Result, ? extends Api.AnyClient>) t)) {
            if (zaj()) {
                t.setFailedResult(new Status(4, (String) null, zak()));
                return t;
            }
            return (T) this.zae.zaa((zaax) t);
        }
        return (T) this.zad.zaa((zaax) t);
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final ConnectionResult zaa(Api<?> api) {
        if (Objects.equal(this.zaf.get(api.zac()), this.zae)) {
            if (zaj()) {
                return new ConnectionResult(4, zak());
            }
            return this.zae.zaa(api);
        }
        return this.zad.zaa(api);
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final void zaa() {
        this.zan = 2;
        this.zal = false;
        this.zak = null;
        this.zaj = null;
        this.zad.zaa();
        this.zae.zaa();
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final ConnectionResult zaa(long j, TimeUnit timeUnit) {
        throw new UnsupportedOperationException();
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final boolean zaa(SignInConnectionListener signInConnectionListener) {
        this.zam.lock();
        try {
            if ((zae() || zad()) && !this.zae.zad()) {
                this.zag.add(signInConnectionListener);
                if (this.zan == 0) {
                    this.zan = 1;
                }
                this.zak = null;
                this.zae.zaa();
                return true;
            }
            this.zam.unlock();
            return false;
        } finally {
            this.zam.unlock();
        }
    }

    private final void zaa(ConnectionResult connectionResult) {
        int i = this.zan;
        if (i != 1) {
            if (i != 2) {
                Log.wtf("CompositeGAC", "Attempted to call failure callbacks in CONNECTION_MODE_NONE. Callbacks should be disabled via GmsClientSupervisor", new Exception());
                this.zan = 0;
            }
            this.zab.zaa(connectionResult);
        }
        zai();
        this.zan = 0;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zaa(int i, boolean z) {
        this.zab.zaa(i, z);
        this.zak = null;
        this.zaj = null;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final void zaa(Bundle bundle) {
        Bundle bundle2 = this.zai;
        if (bundle2 == null) {
            this.zai = bundle;
        } else if (bundle != null) {
            bundle2.putAll(bundle);
        }
    }

    @Override // com.google.android.gms.common.api.internal.zabn
    public final void zaa(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        printWriter.append((CharSequence) str).append("authClient").println(":");
        this.zae.zaa(String.valueOf(str).concat("  "), fileDescriptor, printWriter, strArr);
        printWriter.append((CharSequence) str).append("anonClient").println(":");
        this.zad.zaa(String.valueOf(str).concat("  "), fileDescriptor, printWriter, strArr);
    }
}