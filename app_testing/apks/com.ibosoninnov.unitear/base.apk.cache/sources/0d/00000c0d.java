package com.google.android.datatransport.runtime.backends;

import android.content.Context;
import com.google.android.datatransport.runtime.dagger.internal.Factory;
import d.a.a;

/* loaded from: classes.dex */
public final class MetadataBackendRegistry_Factory implements Factory<MetadataBackendRegistry> {
    private final a<Context> applicationContextProvider;
    private final a<CreationContextFactory> creationContextFactoryProvider;

    public MetadataBackendRegistry_Factory(a<Context> aVar, a<CreationContextFactory> aVar2) {
        this.applicationContextProvider = aVar;
        this.creationContextFactoryProvider = aVar2;
    }

    public static MetadataBackendRegistry_Factory create(a<Context> aVar, a<CreationContextFactory> aVar2) {
        return new MetadataBackendRegistry_Factory(aVar, aVar2);
    }

    public static MetadataBackendRegistry newInstance(Context context, Object obj) {
        return new MetadataBackendRegistry(context, (CreationContextFactory) obj);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.android.datatransport.runtime.dagger.internal.Factory, d.a.a
    public MetadataBackendRegistry get() {
        return newInstance(this.applicationContextProvider.get(), this.creationContextFactoryProvider.get());
    }
}