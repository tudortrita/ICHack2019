import React from 'react';
import { ActivityIndicator,
AsyncStorage,
} from 'react-native';

import {Router, Scene} from 'react-native-router-flux';


import Scanner from './src/screens/Scanner.js';
import Home from './src/screens/Home.js';
import ScannerModal from './src/screens/ScannerModal.js';

export default class App extends React.Component {
    constructor() {
        super();
        this.state = { hasToken: false, isLoaded: false };
    }
    componentDidMount() {
        AsyncStorage.getItem('token').then((token) => {
          this.setState({ hasToken: token !== null, isLoaded: true })
        });
    }
    render() {
        if (!this.state.isLoaded) {
          return (
            <ActivityIndicator />
          )
        } else {
            return (

              <Router>
                    <Scene key='root'>
                        <Scene
                        component={Home}
                        hideNavBar={true}
                        initial={true}
                        key='Home'
                        title='Home'
                        type='reset'
                        initial={!this.state.hasToken}
                        />
                        <Scene
                        component={Scanner}
                        hideNavBar={true}
                        key='Scanner'
                        title='Scanner'
                        type='reset'
                        initial={this.state.hasToken}
                        />


                        <Scene key="ScannerModal" component={ScannerModal} hideNavBar={false} />
                    </Scene>
              </Router>
            );
        }
    }

}
